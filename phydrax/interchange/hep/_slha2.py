#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict SLHA2 mass, mixing, decay, scale, complex, and QNUMBERS semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._slha import SLHABlock, SLHADecay, SLHADocument


@dataclass(frozen=True, slots=True)
class SLHA2Matrix:
    name: str
    scale: float | None
    values: np.ndarray
    unitarity_residual: float
    matrix_id: str


@dataclass(frozen=True, slots=True)
class SLHA2QuantumNumbers:
    pdg_id: int
    three_times_charge: int
    spin_states: int
    color_representation: int
    self_conjugate: bool
    source_block_index: int


@dataclass(frozen=True, slots=True)
class SLHA2SemanticModel:
    masses: tuple[tuple[int, float], ...]
    matrices: tuple[SLHA2Matrix, ...]
    quantum_numbers: tuple[SLHA2QuantumNumbers, ...]
    decays: tuple[SLHADecay, ...]
    scaled_blocks: tuple[tuple[str, float | None, int], ...]
    warning_messages: tuple[str, ...]
    error_messages: tuple[str, ...]
    unknown_blocks: tuple[str, ...]
    source_id: str
    semantic_id: str

    def matrix(self, name: str, /, *, scale: float | None = None) -> SLHA2Matrix:
        target = str(name).upper()
        matches = tuple(
            value
            for value in self.matrices
            if value.name == target and (scale is None or value.scale == scale)
        )
        if len(matches) != 1:
            raise ValueError("SLHA2 matrix is absent or ambiguous across scales.")
        return matches[0]


def _block_matrix(block: SLHABlock, imaginary: SLHABlock | None, /) -> np.ndarray:
    entries = tuple(value for value in block.entries if len(value.indices) == 2)
    if not entries:
        raise ValueError(f"SLHA2 block {block.name} has no matrix entries.")
    rows = max(int(value.indices[0]) for value in entries)
    columns = max(int(value.indices[1]) for value in entries)
    if rows < 1 or columns < 1:
        raise ValueError("SLHA2 matrix indices are one-based positive integers.")
    matrix = np.zeros((rows, columns), dtype=np.complex128)
    seen: set[tuple[int, int]] = set()
    for entry in entries:
        row, column = (int(index) for index in entry.indices)
        if (row, column) in seen:
            raise ValueError("SLHA2 matrix contains a duplicate entry.")
        seen.add((row, column))
        matrix[row - 1, column - 1] = entry.value
    if imaginary is not None:
        for entry in imaginary.entries:
            if len(entry.indices) != 2:
                raise ValueError("Imaginary SLHA2 matrix blocks require two indices.")
            row, column = (int(index) for index in entry.indices)
            if row > rows or column > columns:
                raise ValueError("Imaginary SLHA2 matrix entry leaves its real matrix.")
            matrix[row - 1, column - 1] += 1j * entry.value
    return matrix


def _matching_imaginary_block(
    blocks: Sequence[SLHABlock],
    real_block: SLHABlock,
    /,
) -> SLHABlock | None:
    target = "IM" + real_block.name
    matches = tuple(
        value
        for value in blocks
        if value.name == target and value.scale == real_block.scale
    )
    if len(matches) > 1:
        raise ValueError("SLHA2 imaginary mixing matrix is duplicated at one scale.")
    return matches[0] if matches else None


def interpret_slha2(
    document: SLHADocument,
    /,
    *,
    required_blocks: Sequence[str] = ("MASS",),
    mixing_blocks: Sequence[str] = (
        "NMIX",
        "UMIX",
        "VMIX",
        "STOPMIX",
        "SBOTMIX",
        "STAUMIX",
        "VCKM",
        "UPMNS",
    ),
    unitarity_tolerance: float = 1e-6,
) -> SLHA2SemanticModel:
    """Interpret strict SLHA2 semantics while retaining every raw document block."""

    if not isinstance(document, SLHADocument):
        raise TypeError("document must be SLHADocument.")
    required = tuple(str(value).upper() for value in required_blocks)
    available = {value.name for value in document.blocks}
    missing = tuple(value for value in required if value not in available)
    if missing:
        raise ValueError(f"SLHA2 document is missing required blocks {missing}.")
    mass_blocks = tuple(value for value in document.blocks if value.name == "MASS")
    if len(mass_blocks) != 1:
        raise ValueError("SLHA2 requires one unscaled MASS block.")
    masses = tuple(
        sorted(
            (int(entry.indices[0]), entry.value)
            for entry in mass_blocks[0].entries
            if len(entry.indices) == 1
        )
    )
    if not masses or any(not math.isfinite(value) for _, value in masses):
        raise ValueError("SLHA2 MASS entries must be finite and non-empty.")
    mixing = {str(value).upper() for value in mixing_blocks}
    matrices: list[SLHA2Matrix] = []
    for block in document.blocks:
        if block.name not in mixing:
            continue
        values = _block_matrix(
            block,
            _matching_imaginary_block(document.blocks, block),
        )
        residual = (
            float(np.linalg.norm(values.conj().T @ values - np.eye(values.shape[1])))
            if values.shape[0] == values.shape[1]
            else math.inf
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("SLHA2 mixing matrices must be finite.")
        matrix_id = canonical_fingerprint(
            {
                "kind": "slha2-matrix",
                "name": block.name,
                "scale": block.scale,
                "values": array_tree_fingerprint(values),
                "unitarity_residual": residual,
            }
        )
        matrices.append(SLHA2Matrix(block.name, block.scale, values, residual, matrix_id))
    qnumbers: list[SLHA2QuantumNumbers] = []
    for block_index, block in enumerate(document.blocks):
        if block.name != "QNUMBERS":
            continue
        if len(block.header_arguments) != 1:
            raise ValueError("QNUMBERS headers require exactly one PDG identifier.")
        pdg = int(block.header_arguments[0])
        values = {
            int(entry.indices[0]): round(entry.value)
            for entry in block.entries
            if len(entry.indices) == 1
        }
        if set(values) != {1, 2, 3, 4}:
            raise ValueError("QNUMBERS blocks require entries 1, 2, 3, and 4.")
        qnumbers.append(
            SLHA2QuantumNumbers(
                pdg,
                values[1],
                values[2],
                values[3],
                values[4] == 0,
                block_index,
            )
        )
    warnings: list[str] = []
    errors: list[str] = []
    for block in document.blocks:
        if block.name not in ("SPINFO", "DCINFO"):
            continue
        for entry in block.entries:
            message = entry.comment or entry.raw_value
            if entry.indices and int(entry.indices[0]) == 3:
                warnings.append(message)
            elif entry.indices and int(entry.indices[0]) == 4:
                errors.append(message)
    if any(
        value.name in mixing and value.unitarity_residual > unitarity_tolerance
        for value in matrices
    ):
        warnings.append("one-or-more-mixing-matrices-fail-unitarity-tolerance")
    scaled = tuple(
        (block.name, block.scale, index) for index, block in enumerate(document.blocks)
    )
    semantic_id = canonical_fingerprint(
        {
            "kind": "slha2-semantic-model",
            "source": document.source_id,
            "masses": masses,
            "matrices": [value.matrix_id for value in matrices],
            "quantum_numbers": [
                (
                    value.pdg_id,
                    value.three_times_charge,
                    value.spin_states,
                    value.color_representation,
                    value.self_conjugate,
                )
                for value in qnumbers
            ],
            "scaled_blocks": scaled,
            "warnings": warnings,
            "errors": errors,
        }
    )
    return SLHA2SemanticModel(
        masses,
        tuple(matrices),
        tuple(qnumbers),
        document.decays,
        scaled,
        tuple(warnings),
        tuple(errors),
        document.diagnostics.unknown_block_names,
        document.source_id,
        semantic_id,
    )


__all__ = [
    "SLHA2Matrix",
    "SLHA2QuantumNumbers",
    "SLHA2SemanticModel",
    "interpret_slha2",
]
