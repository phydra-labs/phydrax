#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict clean-room Wannier90 ``*_hr.dat`` bytes adapter."""

from __future__ import annotations

import re

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...operators.periodic._family import (
    periodic_translation_family_from_dense_blocks,
    PeriodicResourceError,
    PreparedPeriodicTranslationFamily,
)
from ..periodic._source import PeriodicSourceContext


_INTEGER = re.compile(r"[+-]?\d+")
_FLOAT = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?")


def _integer(token: str, noun: str) -> int:
    if _INTEGER.fullmatch(token) is None:
        raise ValueError(f"Wannier90 HR {noun} is not an integer.")
    return int(token)


def _float(token: str, noun: str) -> float:
    if _FLOAT.fullmatch(token) is None:
        raise ValueError(f"Wannier90 HR {noun} is not a finite decimal.")
    value = float(token.replace("D", "E").replace("d", "e"))
    if not np.isfinite(value):
        raise ValueError(f"Wannier90 HR {noun} must be finite.")
    return value


class Wannier90HRImport(StrictModule, NonTrainableState):
    context: PeriodicSourceContext
    translations: Array
    degeneracies: Array
    raw_hamiltonian_blocks: Array
    prepared_family: PreparedPeriodicTranslationFamily
    import_id: str = eqx.field(static=True)

    def __init__(
        self,
        context: PeriodicSourceContext,
        translations,
        degeneracies,
        raw_hamiltonian_blocks,
        prepared_family: PreparedPeriodicTranslationFamily,
        /,
    ):
        if not isinstance(context, PeriodicSourceContext) or not isinstance(
            prepared_family, PreparedPeriodicTranslationFamily
        ):
            raise TypeError(
                "HR import requires source context and prepared translation family."
            )
        translation = jnp.asarray(translations, dtype=jnp.int32)
        degeneracy = jnp.asarray(degeneracies, dtype=jnp.int32)
        block = jnp.asarray(raw_hamiltonian_blocks)
        self.context = context
        self.translations = translation
        self.degeneracies = degeneracy
        self.raw_hamiltonian_blocks = block
        self.prepared_family = prepared_family
        self.import_id = canonical_fingerprint(
            {
                "kind": "wannier90-hr-import",
                "context": context.context_id,
                "family": prepared_family.prepared_id,
                "arrays": array_tree_fingerprint(
                    {
                        "translations": np.asarray(translation),
                        "degeneracies": np.asarray(degeneracy),
                        "raw_blocks": np.asarray(block),
                    }
                ),
            }
        )


def read_wannier90_hr(
    payload: bytes,
    context: PeriodicSourceContext,
    /,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
    maximum_orbitals: int = 4096,
    maximum_translations: int = 1_000_000,
    maximum_records: int = 8_000_000,
) -> Wannier90HRImport:
    """Parse only explicitly supplied HR bytes bound to a complete source context."""

    if not isinstance(context, PeriodicSourceContext):
        raise TypeError("Wannier90 HR parsing requires PeriodicSourceContext.")
    context.admit_bytes(payload)
    if len(payload) > int(maximum_bytes):
        raise PeriodicResourceError("Wannier90 HR payload exceeds maximum_bytes.")
    text = payload.decode("utf-8", errors="strict")
    lines = text.splitlines()
    if len(lines) < 4 or not lines[0].strip():
        raise ValueError(
            "Wannier90 HR requires comment, dimensions, degeneracies, and records."
        )
    orbital_count = _integer(lines[1].strip(), "orbital count")
    translation_count = _integer(lines[2].strip(), "translation count")
    if orbital_count != context.basis.orbital_count:
        raise ValueError("Wannier90 HR orbital count does not match source basis order.")
    if orbital_count <= 0 or orbital_count > int(maximum_orbitals):
        raise PeriodicResourceError("Wannier90 HR orbital count exceeds policy.")
    if translation_count <= 0 or translation_count > int(maximum_translations):
        raise PeriodicResourceError("Wannier90 HR translation count exceeds policy.")
    record_count = translation_count * orbital_count * orbital_count
    if record_count > int(maximum_records):
        raise PeriodicResourceError("Wannier90 HR record count exceeds policy.")
    degeneracy_line_count = (translation_count + 14) // 15
    if len(lines) < 3 + degeneracy_line_count + record_count:
        raise ValueError("Wannier90 HR payload is truncated.")
    degeneracy_tokens = " ".join(lines[3 : 3 + degeneracy_line_count]).split()
    if len(degeneracy_tokens) != translation_count:
        raise ValueError("Wannier90 HR degeneracy list is incomplete or overfull.")
    degeneracies = np.asarray(
        [_integer(token, "degeneracy") for token in degeneracy_tokens], dtype=np.int32
    )
    if np.any(degeneracies <= 0):
        raise ValueError("Wannier90 HR degeneracies must be positive.")
    record_lines = lines[3 + degeneracy_line_count :]
    if len(record_lines) != record_count:
        raise ValueError("Wannier90 HR requires exactly nrpts*num_wann^2 records.")
    records: dict[tuple[int, int, int, int, int], complex] = {}
    translation_order: list[tuple[int, int, int]] = []
    for line in record_lines:
        tokens = line.split()
        if len(tokens) != 7:
            raise ValueError("Wannier90 HR records require exactly seven fields.")
        lattice = tuple(_integer(token, "translation") for token in tokens[:3])
        row = _integer(tokens[3], "row index") - 1
        column = _integer(tokens[4], "column index") - 1
        if row < 0 or row >= orbital_count or column < 0 or column >= orbital_count:
            raise ValueError(
                "Wannier90 HR matrix index exceeds the source orbital basis."
            )
        key = lattice + (row, column)
        if key in records:
            raise ValueError(
                "Wannier90 HR contains a duplicate translation/matrix record."
            )
        records[key] = complex(
            _float(tokens[5], "real part"), _float(tokens[6], "imaginary part")
        )
        if lattice not in translation_order:
            translation_order.append(lattice)
    if len(translation_order) != translation_count:
        raise ValueError("Wannier90 HR translation coverage does not match nrpts.")
    if context.basis.cell.rank < 3 and any(
        any(lattice[axis] != 0 for axis in range(context.basis.cell.rank, 3))
        for lattice in translation_order
    ):
        raise ValueError("Wannier90 HR uses translations outside the source cell rank.")
    blocks = np.empty(
        (translation_count, orbital_count, orbital_count), dtype=np.complex128
    )
    for translation_index, lattice in enumerate(translation_order):
        for row in range(orbital_count):
            for column in range(orbital_count):
                key = lattice + (row, column)
                if key not in records:
                    raise ValueError("Wannier90 HR is missing a matrix record.")
                blocks[translation_index, row, column] = records[key]
    translation_lookup = {value: index for index, value in enumerate(translation_order)}
    for index, lattice in enumerate(translation_order):
        reverse_key = tuple(-value for value in lattice)
        if reverse_key not in translation_lookup:
            raise ValueError("Wannier90 HR is missing a reverse translation block.")
        reverse = translation_lookup[reverse_key]
        defect = np.max(np.abs(blocks[reverse] - np.conj(blocks[index].T)), initial=0.0)
        if defect > 1.0e-10 * max(float(np.max(np.abs(blocks[index]), initial=0.0)), 1.0):
            raise ValueError(
                "Wannier90 HR reverse translation blocks are not Hermitian adjoints."
            )
    weighted = blocks / degeneracies[:, None, None]
    rank_translations = np.asarray(translation_order, dtype=np.int32)[
        :, : context.basis.cell.rank
    ]
    prepared = periodic_translation_family_from_dense_blocks(
        rank_translations,
        weighted[:, :, None, :, None],
        maximum_edges=maximum_records,
    )
    return Wannier90HRImport(
        context,
        rank_translations,
        degeneracies,
        blocks,
        prepared,
    )


def write_wannier90_hr(imported: Wannier90HRImport, /) -> bytes:
    """Emit canonical UTF-8 HR bytes without changing imported numerical provenance."""

    if not isinstance(imported, Wannier90HRImport):
        raise TypeError("imported must be Wannier90HRImport.")
    translations = np.asarray(imported.translations)
    if translations.shape[1] < 3:
        translations = np.pad(translations, ((0, 0), (0, 3 - translations.shape[1])))
    blocks = np.asarray(imported.raw_hamiltonian_blocks)
    degeneracies = np.asarray(imported.degeneracies)
    lines = [
        f"Phydrax canonical HR export from {imported.context.provenance.source_id}",
        str(imported.context.basis.orbital_count),
        str(translations.shape[0]),
    ]
    for start in range(0, degeneracies.size, 15):
        lines.append(
            " ".join(str(int(value)) for value in degeneracies[start : start + 15])
        )
    for translation, block in zip(translations, blocks, strict=True):
        for row in range(block.shape[0]):
            for column in range(block.shape[1]):
                value = block[row, column]
                lines.append(
                    f"{int(translation[0]):5d} {int(translation[1]):5d} {int(translation[2]):5d} "
                    f"{row + 1:5d} {column + 1:5d} {value.real: .17e} {value.imag: .17e}"
                )
    return ("\n".join(lines) + "\n").encode("utf-8")


__all__ = ["Wannier90HRImport", "read_wannier90_hr", "write_wannier90_hr"]
