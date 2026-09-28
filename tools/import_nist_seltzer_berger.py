#!/usr/bin/env python3
"""Deterministically import a pinned Seltzer--Berger long-form CSV extract.

The tool never downloads, transcribes, or synthesizes values. It verifies a
caller-pinned SHA-256 digest, canonicalizes caller-prepared rows from the
governed source, and writes a compressed NumPy artifact plus a JSON provenance
sidecar. Required CSV columns are ``material_id``, ``kinetic_energy_ev``,
``photon_fraction``, and ``differential_cross_section``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


NIST_SOURCE_URL = "https://physics.nist.gov/PhysRefData/Star/Text/ref.html"
NIST_TECHNICAL_REFERENCE = (
    "S. M. Seltzer and M. J. Berger, Atomic Data and Nuclear Data Tables 35 "
    "(1986) 345-418; NISTIR 4999 (1993)"
)


class Row(NamedTuple):
    material_id: str
    kinetic_energy_ev: float
    photon_fraction: float
    differential_cross_section: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> tuple[Row, ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = (
            "material_id",
            "kinetic_energy_ev",
            "photon_fraction",
            "differential_cross_section",
        )
        if reader.fieldnames is None or any(
            name not in reader.fieldnames for name in required
        ):
            raise ValueError(f"CSV must contain columns {required!r}.")
        rows = tuple(
            Row(
                str(record["material_id"]).strip(),
                float(record["kinetic_energy_ev"]),
                float(record["photon_fraction"]),
                float(record["differential_cross_section"]),
            )
            for record in reader
        )
    if not rows:
        raise ValueError("NIST import source contains no rows.")
    return rows


def _canonicalize(
    rows: tuple[Row, ...],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    materials = tuple(sorted({row.material_id for row in rows}))
    energies = np.asarray(
        sorted({row.kinetic_energy_ev for row in rows}), dtype=np.float64
    )
    fractions = np.asarray(
        sorted({row.photon_fraction for row in rows}), dtype=np.float64
    )
    if (
        any(not material for material in materials)
        or energies.size < 2
        or fractions.size < 3
        or np.any(~np.isfinite(energies))
        or np.any(energies <= 0.0)
        or np.any(~np.isfinite(fractions))
        or np.any(fractions <= 0.0)
        or np.any(fractions >= 1.0)
    ):
        raise ValueError("NIST table axes are invalid.")
    material_index = {value: index for index, value in enumerate(materials)}
    energy_index = {value: index for index, value in enumerate(energies.tolist())}
    fraction_index = {value: index for index, value in enumerate(fractions.tolist())}
    values = np.full(
        (len(materials), energies.size, fractions.size), np.nan, dtype=np.float64
    )
    for row in rows:
        if (
            not np.isfinite(row.differential_cross_section)
            or row.differential_cross_section < 0.0
        ):
            raise ValueError(
                "Differential cross sections must be finite and nonnegative."
            )
        index = (
            material_index[row.material_id],
            energy_index[row.kinetic_energy_ev],
            fraction_index[row.photon_fraction],
        )
        if np.isfinite(values[index]):
            raise ValueError(f"Duplicate NIST table row at {index!r}.")
        values[index] = row.differential_cross_section
    if np.any(~np.isfinite(values)):
        raise ValueError(
            "NIST table must be a complete material × energy × fraction grid."
        )
    return materials, energies, fractions, values


def import_table(source: Path, output: Path, expected_sha256: str) -> dict[str, object]:
    observed = _sha256(source)
    if observed != expected_sha256.lower():
        raise ValueError(
            f"Source SHA-256 mismatch: expected {expected_sha256}, observed {observed}."
        )
    materials, energies, fractions, values = _canonicalize(_read_rows(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        material_ids=np.asarray(materials, dtype=np.str_),
        kinetic_energy_ev=energies,
        photon_fraction=fractions,
        differential_cross_section=values,
    )
    artifact_sha256 = _sha256(output)
    provenance: dict[str, object] = {
        "artifact_sha256": artifact_sha256,
        "kind": "nist-seltzer-berger-differential-bremsstrahlung",
        "license": "United States public domain; 17 U.S.C. § 105",
        "material_ids": list(materials),
        "nist_source_url": NIST_SOURCE_URL,
        "source_sha256": observed,
        "technical_reference": NIST_TECHNICAL_REFERENCE,
        "value_count": int(values.size),
    }
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    arguments = parser.parse_args()
    provenance = import_table(arguments.source, arguments.output, arguments.sha256)
    print(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
