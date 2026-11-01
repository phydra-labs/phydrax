"""Count cell-validity proofs and compare incremental certificates with full ones."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import phydrax.discretization._cell_geometry_validity as validity_module
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import (
    CellValidityCertificate,
    certify_cell_geometry_validity,
    ValidatedCellValidityParent,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry._mesh_certificates import _same_tree_content


def count_cell_proofs(monkeypatch: pytest.MonkeyPatch, /) -> list[int]:
    """Count proofs of individual cell maps (nonlinear and exact affine)."""
    calls = [0]
    for name in ("_certify_nonlinear_source_map", "_affine_quality"):
        original = getattr(validity_module, name)

        def counted(*args: Any, _original: Callable[..., Any] = original) -> Any:
            calls[0] += 1
            return _original(*args)

        monkeypatch.setattr(validity_module, name, counted)
    return calls


def cell_map_digests(certificate: CellValidityCertificate, /) -> set[bytes]:
    table = np.frombuffer(certificate.cell_map_digests, dtype=np.uint8).reshape(-1, 16)
    return {row.tobytes() for row in table if row.any()}


def assert_incremental_matches_full(
    monkeypatch: pytest.MonkeyPatch,
    geometry: CellMesh | CellGeometrySpec,
    parent: CellValidityCertificate | tuple[CellValidityCertificate, ...],
    /,
    *,
    mesh: CellMesh | None = None,
) -> tuple[CellValidityCertificate, int]:
    """Certify with ``parent``; require the full certificate and return the proof count.

    The count must equal the number of distinct cell maps no parent recorded.
    """
    full = certify_cell_geometry_validity(geometry, mesh=mesh)
    parents = parent if isinstance(parent, tuple) else (parent,)
    validated = tuple(ValidatedCellValidityParent(value) for value in parents)
    calls = count_cell_proofs(monkeypatch)
    incremental = certify_cell_geometry_validity(geometry, mesh=mesh, parent=validated)
    monkeypatch.undo()
    assert incremental.certificate_id == full.certificate_id
    assert incremental.cell_map_digests == full.cell_map_digests
    assert _same_tree_content(incremental, full)
    recorded = set().union(*(cell_map_digests(value) for value in parents))
    assert calls[0] == len(cell_map_digests(full) - recorded)
    return full, calls[0]
