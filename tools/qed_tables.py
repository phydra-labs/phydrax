#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate, record, and verify strong-field QED tables.

Builds the nonlinear Compton and Breit–Wheeler `QEDTable` set for the given
parameters (every process in the ``"averaged"``, ``"positive"`` and
``"negative"`` polarizations used by the polarized models) and writes one
deterministic archive: the table arrays in canonical order (``log_chi``,
``rate_values``, ``rate_slopes``, ``cumulative`` per ``process/polarization``
table) and a JSON record of each table's fingerprint, host evidence
(quadrature, rate and spectrum interpolation errors, monotonicity, truncated
mass) and SHA-256 array digest. ``--check`` rebuilds the tables and refuses an
archive whose fingerprints or digests differ, so a recorded table set can be
compared with an external code's tables or re-verified on another host.

    python tools/qed_tables.py --maximum-chi 100 --output qed-tables.npz
    python tools/qed_tables.py --maximum-chi 100 --check qed-tables.npz
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import get_args

import numpy as np

from phydrax.discretization.pic import QEDProcess, QEDTable, QEDTablePolarization


_ARRAYS = ("log_chi", "rate_values", "rate_slopes", "cumulative")


def build(
    maximum_chi: float, nodes_per_decade: int, spectrum_nodes: int, order: int
) -> dict[str, QEDTable]:
    return {
        f"{process}/{polarization}": QEDTable(
            process,
            maximum_chi=maximum_chi,
            polarization=polarization,
            nodes_per_decade=nodes_per_decade,
            spectrum_nodes=spectrum_nodes,
            quadrature_order=order,
        )
        for process in get_args(QEDProcess)
        for polarization in get_args(QEDTablePolarization)
    }


def _arrays(table: QEDTable) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(table, name)) for name in _ARRAYS}


def _digest(arrays: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for name in _ARRAYS:
        value = np.ascontiguousarray(arrays[name], dtype=np.float64)
        hasher.update(name.encode())
        hasher.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def record(tables: dict[str, QEDTable]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "table_id": table.table_id,
            "process": table.process,
            "polarization": table.polarization,
            "minimum_chi": table.minimum_chi,
            "maximum_chi": table.maximum_chi,
            "rows": table.row_count,
            "spectrum_nodes": table.node_count,
            "quadrature_error": table.quadrature_error,
            "rate_interpolation_error": table.rate_interpolation_error,
            "cdf_row_error": table.cdf_row_error,
            "cdf_node_error": table.cdf_node_error,
            "minimum_cdf_increment": table.minimum_cdf_increment,
            "truncated_mass": table.truncated_mass,
            "sha256": _digest(_arrays(table)),
        }
        for name, table in sorted(tables.items())
    }


def write(path: Path, tables: dict[str, QEDTable]) -> None:
    payload: dict[str, np.ndarray] = {
        "record": np.frombuffer(
            json.dumps(record(tables), sort_keys=True).encode(), dtype=np.uint8
        )
    }
    for table_name, table in sorted(tables.items()):
        for name, value in _arrays(table).items():
            payload[f"{table_name}/{name}"] = value
    buffer = io.BytesIO()
    # numpy's stubs bind the keyword mapping to `allow_pickle`.
    np.savez(buffer, **payload)  # ty: ignore[invalid-argument-type]
    path.write_bytes(buffer.getvalue())


def check(path: Path, tables: dict[str, QEDTable]) -> list[str]:
    """Differences between an archive and freshly built tables (empty: identical)."""
    with np.load(path) as archive:
        stored = json.loads(archive["record"].tobytes().decode())
        arrays = {name: archive[name] for name in archive.files if name != "record"}
    current = record(tables)
    problems = []
    for table_name, entry in current.items():
        previous = stored.get(table_name)
        if previous is None:
            problems.append(f"{table_name}: missing from the archive")
            continue
        for key in ("table_id", "sha256"):
            if previous[key] != entry[key]:
                problems.append(f"{table_name}: {key} differs")
        rebuilt = _digest({name: arrays[f"{table_name}/{name}"] for name in _ARRAYS})
        if rebuilt != previous["sha256"]:
            problems.append(f"{table_name}: archived arrays do not match their digest")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maximum-chi", type=float, required=True)
    parser.add_argument("--nodes-per-decade", type=int, default=16)
    parser.add_argument("--spectrum-nodes", type=int, default=513)
    parser.add_argument("--quadrature-order", type=int, default=8)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--check", type=Path)
    arguments = parser.parse_args()
    tables = build(
        arguments.maximum_chi,
        arguments.nodes_per_decade,
        arguments.spectrum_nodes,
        arguments.quadrature_order,
    )
    if arguments.output is not None:
        write(arguments.output, tables)
        print(json.dumps(record(tables), indent=2, sort_keys=True))
        return
    problems = check(arguments.check, tables)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"{arguments.check}: tables reproduced bitwise.")


if __name__ == "__main__":
    main()
